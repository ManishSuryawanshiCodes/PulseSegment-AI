const fs = require('fs')

const origSymlink = fs.promises.symlink
fs.promises.symlink = async function (target, path, type) {
  try {
    return await origSymlink.call(this, target, path, type)
  } catch (err) {
    if (err.code === 'EPERM') {
      try {
        const stat = await fs.promises.stat(target)
        if (stat.isDirectory()) {
          try {
            return await origSymlink.call(this, target, path, 'junction')
          } catch {
            return await fs.promises.cp(target, path, { recursive: true })
          }
        } else {
          return await fs.promises.copyFile(target, path)
        }
      } catch (e) {
        throw err
      }
    }
    throw err
  }
}

const origSymlinkSync = fs.symlinkSync
fs.symlinkSync = function (target, path, type) {
  try {
    return origSymlinkSync.call(this, target, path, type)
  } catch (err) {
    if (err.code === 'EPERM') {
      try {
        const stat = fs.statSync(target)
        if (stat.isDirectory()) {
          try {
            return origSymlinkSync.call(this, target, path, 'junction')
          } catch {
            return fs.cpSync(target, path, { recursive: true })
          }
        } else {
          return fs.copyFileSync(target, path)
        }
      } catch (e) {
        throw err
      }
    }
    throw err
  }
}
